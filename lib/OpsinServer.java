import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

import uk.ac.cam.ch.wwmm.opsin.NameToStructure;
import uk.ac.cam.ch.wwmm.opsin.NameToStructureConfig;
import uk.ac.cam.ch.wwmm.opsin.OpsinResult;
import uk.ac.cam.ch.wwmm.opsin.OpsinWarning;

/**
 * name2struct2 用の常駐 OPSIN。1 行 = "フラグ<TAB>名前" を読み、1 行で返す。
 * フラグ: l = 立体が読めなくても警告で通す, a = "acetic" のような acid 抜きを許す, c = CML(位置番号)も返す
 * 返り: 状態<TAB>SMILES<TAB>警告<TAB>メッセージ<TAB>CML(改行なし)
 */
public class OpsinServer {
    public static void main(String[] args) throws Exception {
        NameToStructure n2s = NameToStructure.getInstance();
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8), false);
        out.println("READY\t" + NameToStructure.getVersion());
        out.flush();
        String line;
        while ((line = in.readLine()) != null) {
            int tab = line.indexOf('\t');
            String flags = tab < 0 ? "" : line.substring(0, tab);
            String name = tab < 0 ? line : line.substring(tab + 1);
            NameToStructureConfig cfg = new NameToStructureConfig();
            cfg.setDetailedFailureAnalysis(true);
            cfg.setWarnRatherThanFailOnUninterpretableStereochemistry(flags.contains("l"));
            cfg.setInterpretAcidsWithoutTheWordAcid(flags.contains("a"));
            String status, smi = "", warn = "", msg = "", cml = "";
            try {
                OpsinResult r = n2s.parseChemicalName(name, cfg);
                status = r.getStatus().toString();
                if (r.getStatus() != OpsinResult.OPSIN_RESULT_STATUS.FAILURE) {
                    smi = r.getSmiles();
                    if (smi == null) smi = "";
                    StringBuilder w = new StringBuilder();
                    for (OpsinWarning ow : r.getWarnings()) {
                        if (w.length() > 0) w.append(" | ");
                        w.append(ow.getType()).append(": ").append(ow.getMessage());
                    }
                    warn = w.toString();
                    if (flags.contains("c")) {
                        String c = r.getCml();
                        cml = c == null ? "" : c;
                    }
                }
                msg = r.getMessage() == null ? "" : r.getMessage();
            } catch (Exception e) {
                status = "FAILURE";
                msg = e.toString();
            }
            out.println(status + "\t" + clean(smi) + "\t" + clean(warn) + "\t" + clean(msg) + "\t" + clean(cml));
            out.flush();
        }
    }

    private static String clean(String s) {
        return s.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ');
    }
}
